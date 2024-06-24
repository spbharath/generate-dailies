import os, sys, re
from glob import glob
import time, datetime
import logging
import argparse, shlex
import subprocess

from tc import Timecode
import pyseq
import json

try:
    import oiio

    # import OpenImageIO as oiio
    import numpy as np
    import yaml
    from PIL import Image
except ImportError:
    print(
        "Error: Missing dependencies. Need:\n\tOpenImageIO\n\tNumPy\n\tPyYAML\n\tPillow (for mjpeg codec conversion)"
    )


dir_path = os.path.dirname(os.path.realpath(__file__))

DAILIES_CONFIG_DEFAULT = os.path.join(dir_path, "dailies-config.json")
DEFAULT_CODEC = "hevc"
DEFAULT_OCIO_FILE_PATH = "./config.ocio"
DEFAULT_OCIO_TRANSFORM = ["linear", "sRGB"]

log = logging.getLogger(__name__)


class GenrateDaily:
    def __init__(self):
        self.start_time = time.time()
        self.setup_success = False
        self.renamed_file = ""
        self.first_frame_path = ""

        # Parse Config File
        DAILIES_CONFIG = os.getenv("DAILIES_CONFIG")
        if not DAILIES_CONFIG:
            DAILIES_CONFIG = DAILIES_CONFIG_DEFAULT

        # Get Config file data
        if os.path.isfile(DAILIES_CONFIG):
            with open(DAILIES_CONFIG, "r") as configfile:
                config = json.load(configfile)
        else:
            print("Error: Could not find config file {0}".format(DAILIES_CONFIG))
            self.setup_success = False
            return

        output_codecs = config.get("output_codecs").keys()
        self.slate_profile = config.get("slate_profiles")
        ocio_profiles = config.get("ocio_profiles")

        parser = argparse.ArgumentParser(
            description="Process given image sequence with ocio display, resize and output to ffmpeg for encoding into a dailies movie."
        )
        parser.add_argument(
            "input_path",
            help="Input exr image sequence. Can be a folder containing images, a path to the first image, a percent 05d path, or a ##### path.",
        )

        parser.add_argument(
            "-o",
            "--output",
            help="Output directory: Optional override to movie_location in the DAILIES_CONFIG. This can be a path relative to the image sequence.",
        )
        parser.add_argument(
            "-d", "--debug", help="Set debug to true.", action="store_true"
        )

        # Show help if no args.
        if len(sys.argv) == 1:
            parser.print_help()
            return None

        args = parser.parse_args()

        self.globals_config = config.get("globals")
        input_path = args.input_path
        codec = self.globals_config.get("output_codec")
        self.movie_location = args.output

        if args.debug:
            print("Setting DEBUG=True!")
            DEBUG = True

        if not codec:
            codec = DEFAULT_CODEC

        self.codec_config = config["output_codecs"][codec]

        self.image_sequences = self.get_image_sequences(input_path)

        if not self.image_sequences:
            print("No image sequence found! Exiting...")
            self.setup_success = False
            return

        self.ocioconfig = self.globals_config.get("ocioconfig")
        if self.ocioconfig:
            log.debug("Got OCIO config from config: {0}".format(self.ocioconfig))
        # Try to get ocio config from $OCIO env-var if it's not defined
        if not self.ocioconfig:
            self.ocioconfig = DEFAULT_OCIO_FILE_PATH

        if not os.path.exists(self.ocioconfig):
            log.warning(
                "OCIO Config does not exist: \n\t{0}\n\tNo OCIO color transform will be applied".format(
                    self.ocioconfig
                )
            )
            self.ocioconfig = None

        # Get default ocio transform to use if none is passed by commandline
        self.ociocolorconvert = self.globals_config.get("ocio_transform")

        if self.ociocolorconvert:
            log.debug(
                "Got OCIO Transform from config: {0}".format(self.ociocolorconvert)
            )
        # Try to get ocio config from $OCIO env-var if it's not defined
        if not self.ocioconfig:
            self.ocioconfig = DEFAULT_OCIO_TRANSFORM
        else:
            # No ocio color transform specified
            print(
                "Warning: No default ocio transform specified, and no transform specified. No color transform will occur."
            )
            self.ociocolorconvert = None

        self.output_width = self.globals_config["width"]
        self.output_height = self.globals_config["height"]

        if not self.output_width or not self.output_height:
            buf = oiio.ImageBuf(self.image_sequence[0].path)
            spec = buf.spec()
            iar = float(spec.width) / float(spec.height)
            if not self.output_width:
                self.output_width = spec.width
                self.globals_config["width"] = self.output_width
            if not self.output_height:
                self.output_height = int(round(self.output_width / iar))
                self.globals_config["height"] = self.output_height
            buf.close()

        self.setup_success = True

        if self.setup_success == True:
            for self.image_sequence in self.image_sequences:
                self.process()

    def process(self):
        """
        Performs the actual processing of the movie.
        Args:
            None
        Returns:
            None
        """

        # Set up movie file location and naming

        # Crop separating character from sequence basename if there is one.
        seq_basename = self.image_sequence.head()

        if seq_basename.endswith(self.image_sequence.parts[-2]):
            seq_basename = seq_basename[:-1]

        movie_ext = self.globals_config["movie_ext"]

        # Create full movie filename
        # Append codec to dailies movie name if requested
        if self.globals_config["movie_append_codec"]:
            codec_name = self.codec_config.get("name")
            if not codec_name:
                print("No codec name! Please fix the config!")
                print(self.codec_config)
                codec_name = ""
            else:
                movie_basename = seq_basename + "_" + codec_name
                movie_filename = movie_basename + "." + movie_ext
        else:
            movie_basename = seq_basename
            movie_filename = seq_basename + "." + movie_ext

        # Handle relative / absolute paths for movie location
        # use globals config for movie location if none specified on the commandline
        if not self.movie_location:
            self.movie_location = self.globals_config["movie_location"]
            print(
                "No output folder specified. Using Output folder from globals: {0}".format(
                    self.movie_location
                )
            )

        if self.movie_location.startswith("/"):
            # Absolute path specified
            self.movie_fullpath = os.path.join(self.movie_location, movie_filename)
        elif self.movie_location.startswith("~"):
            # Path referencing home folder specified
            self.movie_location = os.path.expanduser(self.movie_location)
            self.movie_fullpath = os.path.join(self.movie_location, movie_filename)
        elif self.movie_location.startswith(".") or self.movie_location.startswith(
            ".."
        ):
            # Relative path specified - will output relative to image sequence directory
            self.movie_fullpath = os.path.join(
                self.image_sequence.dirname, self.movie_location, movie_filename
            )
        else:
            self.movie_fullpath = os.path.join(self.movie_location, movie_filename)

        # Check output dir exists
        if not os.path.exists(os.path.dirname(self.movie_fullpath)):
            try:
                os.makedirs(os.path.dirname(self.movie_fullpath))
            except OSError:
                print(
                    "Output directory does not exist and do not have permission to create it: \n\t{0}".format(
                        os.path.dirname(self.movie_fullpath)
                    )
                )
                return

        # Set up Logger
        log_fullpath = os.path.splitext(self.movie_fullpath)[0] + ".log"
        if os.path.exists(log_fullpath):
            os.remove(log_fullpath)
        handler = logging.FileHandler(log_fullpath)
        handler.setFormatter(
            logging.Formatter(
                "%(levelname)s\t %(asctime)s \t%(message)s", "%Y-%m-%dT%H:%M:%S"
            )
        )
        log.addHandler(handler)
        if self.globals_config["debug"]:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(
            "Got config:\n\tCodec Config:\t{0}\n\tImage Sequence Path:\n\t\t{1}".format(
                self.codec_config["name"], self.image_sequence.path()
            )
        )

        log.debug(
            "Output width x height: {0}x{1}".format(
                self.output_width, self.output_height
            )
        )

        # Set pixel_data_type based on config bitdepth
        if self.codec_config["bitdepth"] > 8:
            self.pixel_data_type = oiio.UINT16
        else:
            self.pixel_data_type = oiio.UINT8

        # Get timecode based on frame
        tc = Timecode(self.globals_config["framerate"], start_timecode="00:00:00:00")
        self.start_tc = tc + self.image_sequence.start()

        ffmpeg_args = self.setup_ffmpeg()

        log.info("ffmpeg command:\n\t{0}".format(ffmpeg_args))

        # Static image buffer for text that doesn't change frame to frame
        self.static_text_buf = oiio.ImageBuf(
            oiio.ImageSpec(
                self.output_width, self.output_height, 4, self.pixel_data_type
            )
        )

        self.zero_frame = self.slate_profile.get("zero_frame")

        # Loop through each text element, create the text image, and add it to self.static_text_buf
        zero_frame_text_elements = self.zero_frame.get("static_text_elements")
        if zero_frame_text_elements:
            for text_element_name, text_element in zero_frame_text_elements.items():
                log.info("Generate Text")
                self.generate_text(
                    text_element_name, text_element, self.static_text_buf
                )

    def setup_ffmpeg(self):
        """
        Constructs an ffmpeg command based on the given codec config.

        Returns:
            A string containing the entire ffmpeg command to run.
        """

        # ffmpeg-10bit No longer necessary in ffmpeg > 4.1
        ffmpeg_command = "ffmpeg"

        if self.codec_config["bitdepth"] >= 10:
            pixel_format = "rgb48le"
        else:
            pixel_format = "rgb24"

        if self.codec_config["name"] == "mjpeg":
            # Set up input arguments for frame input through pipe:
            args = "{0} -y -framerate {1} -i pipe:0".format(
                ffmpeg_command, self.globals_config["framerate"]
            )
        else:
            # Set up input arguments for raw video and pipe:
            args = "{0} -hide_banner -loglevel info -y -f rawvideo -pixel_format {1} -video_size {2}x{3} -framerate {4} -i pipe:0".format(
                ffmpeg_command,
                pixel_format,
                self.globals_config["width"],
                self.globals_config["height"],
                self.globals_config["framerate"],
            )

        # Add timecode so that start frame will display correctly in RV etc
        args += " -timecode {0}".format(self.start_tc)

        if self.codec_config["codec"]:
            args += " -c:v {0}".format(self.codec_config["codec"])

        if self.codec_config["profile"]:
            args += " -profile:v {0}".format(self.codec_config["profile"])

        if self.codec_config["qscale"]:
            args += " -qscale:v {0}".format(self.codec_config["qscale"])

        if self.codec_config["preset"]:
            args += " -preset {0}".format(self.codec_config["preset"])

        if self.codec_config["keyint"]:
            args += " -g {0}".format(self.codec_config["keyint"])

        if self.codec_config["bframes"]:
            args += " -bf {0}".format(self.codec_config["bframes"])

        if self.codec_config["tune"]:
            args += " -tune {0}".format(self.codec_config["tune"])

        if self.codec_config["crf"]:
            args += " -crf {0}".format(self.codec_config["crf"])

        if self.codec_config["pix_fmt"]:
            args += " -pix_fmt {0}".format(self.codec_config["pix_fmt"])

        if self.globals_config["framerate"]:
            args += " -r {0}".format(self.globals_config["framerate"])

        if self.codec_config["vf"]:
            args += " -vf {0}".format(self.codec_config["vf"])

        if self.codec_config["vendor"]:
            args += " -vendor {0}".format(self.codec_config["vendor"])

        if self.codec_config["metadata_s"]:
            args += " -metadata:s {0}".format(self.codec_config["metadata_s"])

        if self.codec_config["bitrate"]:
            args += " -b:v {0}".format(self.codec_config["bitrate"])

        # Finally add the output movie file path
        args += " {0}".format(self.movie_fullpath)

        return args

    def get_image_sequences(self, input_path):
        """
        Get list of image sequence objects given a path on disk.

        Args:
            input_path: Input file path. Can be a directory or file or %05d / ### style

        Returns:
            An image sequence object.
        """
        input_path = os.path.realpath(input_path)
        input_image_formats = [
            "exr",
            "tif",
            "tiff",
            "png",
            "jpg",
            "jpeg",
            "iff",
            "tex",
            "tx",
            "jp2",
            "j2c",
        ]
        print("Processing INPUT PATH: {0}".format(input_path))
        if os.path.isdir(input_path):
            # Find image sequences recursively inside specified directory
            # self.create_temp_frame(input_path)
            image_sequences = []
            for root, directories, filenames in os.walk(input_path):
                # If there is more than 1 image file in input_path, search this path for file sequences also
                if root == input_path:
                    image_files = [
                        f
                        for f in filenames
                        if os.path.splitext(f)[-1][1:] in input_image_formats
                    ]
                    if len(image_files) > 1:
                        image_sequences += pyseq.get_sequences(input_path)
                for directory in directories:
                    image_sequences += pyseq.get_sequences(
                        os.path.join(root, directory)
                    )
            if not image_sequences:
                log.error(
                    "Could not find any image files recursively in source directory: {0}".format(
                        input_path
                    )
                )
                return None
        elif os.path.isfile(input_path):
            # Assume it's the first frame of the image sequence
            # Try to split off the frame number to get a glob
            image = pyseq.get_sequences(input_path)
            if image:
                image = image[0]
            image_sequences = pyseq.get_sequences(
                os.path.join(image.dirname, image.name.split(image.parts[-2])[0]) + "*"
            )

        else:
            # Assume this is a %05d or ### image sequence. Use the parent directory if it exists.
            dirname, filename = os.path.split(input_path)
            if os.path.isdir(dirname):
                image_sequences = pyseq.get_sequences(dirname)
            else:
                image_sequences = None

        if image_sequences:
            # Remove image sequences not in list of approved extensions
            if not input_image_formats:
                input_image_formats = ["exr"]
            actual_image_sequences = []
            for image_sequence in image_sequences:
                extension = image_sequence.name.split(".")[-1]
                if extension in input_image_formats:
                    actual_image_sequences.append(image_sequence)
            print("Found image sequences: \n{0}".format(actual_image_sequences))
            return actual_image_sequences
        else:
            log.error("Could not find any Image Sequences!!!")
            return None

    def create_temp_frame(self, input_path):
        import random

        output_width = self.globals_config["width"]
        output_height = self.globals_config["height"]

        image_buf = oiio.ImageBuf(
            oiio.ImageSpec(output_width, output_height, 3, oiio.FLOAT)
        )

        for y in range(output_height):
            for x in range(output_width):
                image_buf.setpixel(x, y, [0.0, 0.0, 0.0])

        f = os.listdir(input_path)
        source_file = os.path.join(input_path, f[random.randint(1, len(f))])
        dir_path, base_filename = os.path.split(f[1])
        base_filename_without_ext, ext = os.path.splitext(base_filename)
        zero_frame_filename = f"{base_filename_without_ext[:-4]}0000{ext}"
        first_frame_filename = f"{base_filename_without_ext[:-4]}0001{ext}"
        self.first_frame_path = os.path.join(input_path, first_frame_filename)
        self.renamed_file = os.path.join(input_path, zero_frame_filename)
        image_buf.write(self.renamed_file)


if __name__ == "__main__":
    daily = GenrateDaily()
