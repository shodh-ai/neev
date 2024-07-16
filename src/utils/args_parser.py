import os
from .config_parser import ConfigReader, Config


def argument_exists(parser, arg):
    return arg in parser._option_string_actions


def training_parser(parser):
    if not argument_exists(parser, "--config"):
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            help="Path to the configuration file",
        )
    if not argument_exists(parser, "--input"):
        parser.add_argument(
            "-i",
            "--input",
            type=str,
            help="Folder containing the data files (*.txt)",
        )
    if not argument_exists(parser, "--output"):
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            help="Folder to save the output files",
        )
    args = parser.parse_args()

    if args.config and args.config.startswith("."):
        args.config = os.path.abspath(args.config)

    if args.config and args.config.startswith(".."):
        args.config = os.path.abspath(
            os.path.join(os.path.dirname(args.config), "..", args.config)
        )

    if args.input and args.input.startswith("."):
        args.input = os.path.abspath(args.input)

    if args.input and args.input.startswith(".."):
        args.input = os.path.abspath(
            os.path.join(os.path.dirname(args.input), "..", args.input)
        )

    if args.output and args.output.startswith("."):
        args.output = os.path.abspath(args.output)

    if args.output and args.output.startswith(".."):
        args.output = os.path.abspath(
            os.path.join(os.path.dirname(args.output), "..", args.output)
        )

    if not args.config and not args.input and not args.output:
        parser.error("Either --config or --input and --output must be specified")

    if args.config and (args.input or args.output):
        parser.error("Either --config or --input and --output must be specified")

    if (args.input and not args.output) or (args.output and not args.input):
        parser.error("Both --input and --output must be specified together")

    if args.config and not os.path.isfile(args.config):
        parser.error(f"Config file not found: {args.config}")

    if args.input and not os.path.isdir(args.input):
        parser.error(f"Input folder not found: {args.input}")

    if args.output and not os.path.isdir(args.output):
        parser.error(f"Output folder not found: {args.output}")

    if args.config:
        config = ConfigReader(args.config)
        args = config.read()
        return args

    if args.input and args.output:
        config = Config()
        config.input = args.input
        config.output = args.output
        config.verify()
        args = config
        return args
