import click
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

from .config import Config, ConfigError
from .converter import AudioConverter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--input', '-i',
    required=True,
    type=click.Path(exists=True),
    help='Input file or directory'
)
@click.option(
    '--output', '-o',
    default='./audiobooks',
    type=click.Path(),
    help='Output directory for audio files'
)
@click.option(
    '--config', '-c',
    default='config.yaml',
    type=click.Path(exists=True),
    help='Configuration file path'
)
@click.option(
    '--format',
    type=click.Choice(['pdf', 'md', 'all']),
    default='all',
    help='File format to process'
)
@click.option(
    '--voice',
    type=str,
    default=None,
    help='Voice type for TTS (uses config.yaml default if not specified)'
)
@click.option(
    '--no-resume',
    is_flag=True,
    help='Do not resume from previous progress'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def main(
    input: str,
    output: str,
    config: str,
    format: str,
    voice: str,
    no_resume: bool,
    verbose: bool
):
    """
    TTS Audiobook Generator - Convert PDF and Markdown files to MP3 audio.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        try:
            cfg = Config.load(config)
        except ConfigError:
            logger.info("Config file not found, using defaults with environment variables")
            cfg = Config.load_defaults()

        # Override with CLI options
        if voice is not None:
            cfg.tts_voice = voice

        # Validate config
        cfg.validate()

        # Collect input files
        input_path = Path(input)
        input_files = []

        if input_path.is_file():
            input_files = [str(input_path)]
        elif input_path.is_dir():
            extensions = {
                'pdf': ['.pdf'],
                'md': ['.md', '.markdown'],
                'all': ['.pdf', '.md', '.markdown']
            }
            valid_exts = extensions[format]

            for ext in valid_exts:
                input_files.extend([str(f) for f in input_path.rglob(f'*{ext}')])

        if not input_files:
            logger.warning(f"No files found in {input}")
            return

        logger.info(f"Found {len(input_files)} files to convert")

        # Run conversion
        converter = AudioConverter(cfg, work_dir=output)

        async def run_conversion():
            results = []
            for file in input_files:
                try:
                    result = await converter.convert_file(
                        file,
                        resume=not no_resume
                    )
                    results.append(result)
                    logger.info(f"✓ Converted: {file} -> {result}")
                except Exception as e:
                    logger.error(f"✗ Failed: {file} - {e}")

            return results

        results = asyncio.run(run_conversion())

        # Summary
        logger.info(f"\nConversion complete!")
        logger.info(f"Successfully converted: {len([r for r in results if r])}/{len(input_files)}")

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please set ALIYUN_API_KEY environment variable or create config.yaml")
        raise click.Abort()
    except Exception as e:
        logger.error(f"Error: {e}")
        raise click.Abort()


if __name__ == '__main__':
    main()
