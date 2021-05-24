#!/usr/bin/env python3

# by TheTechromancer

# This was adapted from the Password Statistical Analysis tool by Peter Kacherginsky
# https://github.com/iphelix/pack

import re
import sys
import argparse
import webbrowser
from pathlib import Path
from .lib.report import *
from password_stretcher.lib.utils import *
from password_stretcher.lib.errors import *
from password_stretcher.lib.policy import PasswordPolicy


def main():

    parser = argparse.ArgumentParser(description='PASSWORDS NED, IN AN OPEN FIELD')
    parser.add_argument('-i', '--input', nargs='+', type=Path, default=ReadSTDIN(binary=False), help='password list(s) to analyze (default: STDIN)', metavar='')
    parser.add_argument('-o', '--output', type=Path, help='save all data to this directory (Markdown, XLSX, JSON, PNG)')
    reporting = argparse.ArgumentParser.add_argument_group(parser, 'report options')
    reporting.add_argument('--limit', type=int, default=20, metavar=20, help='limit the number of results per chart (default: 20)')
    reporting.add_argument('--hiderare', type=float, default=5.0, metavar='1.0', help='hide statistics covering less than this percent of the max value (default: 5.0)')
    reporting.add_argument('--title', default='Password Analysis', help='title of report')
    reporting.add_argument('--theme', choices=['light', 'dark'], default='dark', help='visual theme for report (default: dark)')
    filters = argparse.ArgumentParser.add_argument_group(parser, 'password complexity filters')
    filters.add_argument('--minlength', type=int, metavar='8', help='minimum password length')
    filters.add_argument('--maxlength', type=int, metavar='8', help='maximum password length')
    filters.add_argument('--mincharsets', type=int, metavar='3', help='must have this many character sets')
    filters.add_argument('--charsets', nargs='+', choices=PasswordPolicy.charset_choices, help='must include these character sets')
    filters.add_argument('--regex', type=re.compile, metavar='\'$[a-z]*^\'', help='custom regex')
    parsing = argparse.ArgumentParser.add_argument_group(parser, 'password file parsing options')
    parsing.add_argument('-d', '--delimiter', metavar=':', help='file delimiter')
    parsing.add_argument('-f', '--field', default='2-', metavar='2-', help='password field number (default: "2-")')
    webserver = argparse.ArgumentParser.add_argument_group(parser, 'web server options options')
    webserver.add_argument('--port', type=int, default=8050, metavar='80', help='port to listen on (default: 8050)')
    webserver.add_argument('--host', default='127.0.0.1', metavar='0.0.0.0', help='interface to listen on (default: 127.0.0.1)')
    webserver.add_argument('--no-server', action='store_true', help='don\'t start a web server')
    webserver.add_argument('--no-browser', action='store_true', help='don\'t attempt to open a web browser')

    try:

        options = parser.parse_args()

        # print help if there's nothing to analyze
        if type(options.input) == ReadSTDIN and sys.stdin.isatty():
            parser.print_help()
            sys.stderr.write('\n\n[!] Please specify password list(s) or pipe to STDIN\n')
            exit(2)

        if not type(options.input) == ReadSTDIN:
            print(f'[+] Analyzing passwords in {", ".join([str(f) for f in options.input])}')
            options.input = ReadFiles(*options.input, binary=False)

        report = PasswordReport(options)
        report.stats.analyze(options.input)

        if options.output:
            markdown = report.dump_everything(options.output)
        else:
            markdown = report.dump_everything(options.output, write=False)
        print(markdown)

        report.make_html_report(show=not options.no_server)


    except re.error:
        print(f'[!] Invalid regex')
        print('[*] Remember to place regex in single quotes: \'^Password[0-9]+\'')
        sys.exit(2)
    except BrokenPipeError:
        # Python flushes standard streams on exit; redirect remaining output
        # to devnull to avoid another BrokenPipeError at shutdown
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)  # Python exits with error code 1 on EPIPE
    except (PasswordStretcherError, AssertionError) as e:
        sys.stderr.write(f'\n[!] {e}\n')
        exit(1)
    except argparse.ArgumentError:
        sys.stderr.write('\n[!] Check your syntax. Use -h for help.\n')
        exit(2)
    except KeyboardInterrupt:
        sys.stderr.write('\n[!] Interrupted.\n')
        exit(2)


if __name__ == '__main__':
    main()