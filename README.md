![password-smelter](https://user-images.githubusercontent.com/20261699/117364387-d7d9e280-ae8b-11eb-927e-c6d4cbf7b76e.png)

**Analyze the shite out of some passwords.** Compliments [password-stretcher](https://github.com/thetechromancer/password-stretcher).

Ingests passwords from hashcat, etc. and outputs to HTML, Markdown, XLSX, PNG, JSON. Dark and light themes supported.

## Installation
~~~
$ pip install password-smelter
~~~

<br>

![password_smelter](https://user-images.githubusercontent.com/20261699/117196051-05535d00-adb4-11eb-8c5b-56f73a3c0fbc.png)

<br>

## Example: Analyze cracked passwords from hashcat
~~~
$ password-smelter -d: -i <(hashcat --show -m 0 hashes.txt) <(hashcat --left -m 0 hashes.txt)
~~~

## Example: Analyze wordlist from STDIN
~~~
$ cat passwords.txt | password-smelter --minlength 8 --mincharsets 2
~~~

## Example: Output to Markdown, XLSX, PNG, JSON
~~~
$ cat passwords.txt | password-smelter -o analyzed_passwords

$ ls -R analyzed_passwords
analyzed_passwords/:
20210505_150937_password_analysis_images  20210505_150937_password_analysis.md
20210505_150937_password_analysis.json    20210505_150937_password_analysis.xlsx

analyzed_passwords/20210505_150937_password_analysis_images:
advancedmasks.png  basewords.png  charactersets.png  entropy.png  length.png  mutations.png  numbers.png  simplemasks.png  symbols.png
~~~

## Usage:
~~~
$ password-smelter --help
usage: melter.py [-h] [-i  [...]] [-o OUTPUT] [--limit 20] [--hiderare 1.0] [--title TITLE] [--theme {light,dark}] [--minlength 8] [--maxlength 8] [--mincharsets 3]
                 [--charsets {numeric,loweralpha,upperalpha,special} [{numeric,loweralpha,upperalpha,special} ...]] [--regex '$[a-z]*^'] [-d :] [-f 2] [--port 80] [--host 0.0.0.0]
                 [--no-server] [--no-browser]

PASSWORDS NED, IN AN OPEN FIELD

optional arguments:
  -h, --help            show this help message and exit
  -i  [ ...], --input  [ ...]
                        password list(s) to analyze (default: STDIN)
  -o OUTPUT, --output OUTPUT
                        save all data to this directory (it will be created)

report options:
  --limit 20            limit the number of results per chart (default: 20)
  --hiderare 1.0        hide statistics covering less than this percent of the max value (default: 5.0)
  --title TITLE         title of report
  --theme {light,dark}  visual theme for report (default: dark)

password complexity filters:
  --minlength 8         minimum password length
  --maxlength 8         maximum password length
  --mincharsets 3       must have this many character sets
  --charsets {numeric,loweralpha,upperalpha,special} [{numeric,loweralpha,upperalpha,special} ...]
                        must include these character sets
  --regex '$[a-z]*^'    custom regex

password file parsing options:
  -d :, --delimiter :   file delimiter
  -f 2, --field 2       password field number (default: 2)

web server options options:
  --port 80             port to listen on (default: 8050)
  --host 0.0.0.0        interface to listen on (default: 127.0.0.1)
  --no-server           don't start a web server
  --no-browser          don't attempt to open a web browser
~~~

## Credit
Parts of this tool were adapted from the Password Statistical Analysis tool by Peter Kacherginsky
https://github.com/iphelix/pack