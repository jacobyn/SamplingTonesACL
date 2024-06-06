# Installation instructions

Note: if you are planning to develop your own PsyNet experiments, we recommend exploring the 
[PsyNet documentation website](https://psynet.dev) first and following the generic 
installation instructions listed there.

## Prerequisites

This experiment should be compatible with any modern computer running MacOS, Linux, or Windows.
There are four main prerequisites:

- WSL Ubuntu (only required for Windows)
- Docker
- PyCharm (optional)
- Git (optional)

All are free and can be downloaded via the Internet. We will detail how to install each of these in turn.
We will make reference to running commands in your terminal; on MacOS and Linux this is a software
application called Terminal, whereas on Windows this is a software application called PowerShell.

### WSL Ubuntu (only required for Windows)

*MacOS and Linux users can skip this section.*

If you are using Windows you need to install Linux via the Windows Subsystem for Linux (WSL).
You can do this following [these instructions](https://learn.microsoft.com/en-us/windows/wsl/install).
By default, the Ubuntu distribution of Linux should be installed.
You should make sure that Ubuntu is registered as the default WSL distribution.
To check this, run the following in your terminal:

```
wsl --list --all
```

To set Ubuntu as your default distribution:

```
wsl --setdefault Ubuntu
```

Restart your computer before continuing with the next steps.

### Docker

Download Docker Desktop from the [Docker website](https://docs.docker.com/get-docker/) and follow the provided
installation instructions. Installing Docker is typically trouble-free on MacOS and Linux but can be
more complex on Windows. If you run into issues see the Troubleshooting section below.

### PyCharm

Writing code usually benefits from an integrated development environment (IDE). 
We recommend using PyCharm for PsyNet experiments, specifically the Professional Edition.
This is paid but academics can get an educational license for free.
Download PyCharm from [this link](https://www.jetbrains.com/help/pycharm/installation-guide.html).

*Windows users only*: You should configure PyCharm to use Unix-style line endings (LF) by default instead
of Windows-style line endings (CLRF); otherwise your Docker scripts may not run.
To do this, follow
[these instructions from the JetBrains website](https://www.jetbrains.com/help/pycharm/configuring-line-endings-and-line-separators.html>):

  1. Open PyCharm's settings.
  2. Go to File | New Projects Setup | Settings (or Preferences) for New Projects | Editor | Code Style.
  3. Set Line separator to 'Unix and macOS (\n)'.
  4. If you are in a project already, you may wish to select the current project from the Scheme dropdown menu on this
     same page and repeat the process of setting the line seperator.
  5. Press OK.

### Git 

If you want to develop your own experiment it's a good idea to use Git for version control.
You can download Git [here](https://git-scm.com/downloads).
You will probably want to set up a free account with [GitHub](http://github.com/) or similar
to store your repositories online.

*Windows users only*: once you've installed Git, you need to run a few commands in your terminal:

```shell
git config --global core.autocrlf false
git config --global core.eol lf
```

## Downloading the repository

The easiest way to download the code is as a zip file. If you are viewing the repository 
online you should see a link to do this on the web page.

If you want to work on the experiment yourself you should probably download it using Git.
If you are viewing the repository online you should see button saying 'Clone' or similar;
this will give you some download links to copy. You can use these in your terminal.
We recommend you use the 'HTTPS' link.

```shell
# Navigate to the parent directory where you want to download your project.
# The project will be downloaded as a subdirectory within this directory,
# defaulting to the name of the repository.
# Note: you should create the parent directory first if it doesn't exist yet.
cd ~/Documents/psynet-projects

# Clone the Git repository, replacing the URL below with the one you get from
# the website under the Clone with HTTPS option.
git clone https://gitlab.com/pmcharrison/example-experiment.git
```

If the experiment is a private repository then someone should have added you already
as a collaborator. You will need to use your credentials when cloning the repository;
if you use the HTTPS link then you should be prompted for these automatically.

## Setting up PyCharm

The first time you open PyCharm you may need to enter some license information,
decide to start a free trial, or something similar. Do this first.

Now, within PyCharm, click File > Open and open the folder that Git downloaded for you.
This opens the experiment directory as a PyCharm 'project'.
It may ask you to setup an 'interpreter' at this point; ignore this message and click Cancel.

The first thing you should do is 'build' the experiment. The first time you build a PsyNet
experiment it will download PsyNet and lots of other dependencies. Make sure you have a
good internet connection for this, it will take a few minutes.
You build the experiment by running the following in your PyCharm terminal:

```shell
bash docker/build
```

Note: if you see an error message like this:

```shell
./docker/run: Permission denied
```

run the following command, then try again:

```shell
chmod +x docker/*
```

If you see other error messages at this point, see Troubleshooting.

## Running the experiment

If all has gone well, you should now be able to run the experiment. 
Try this by running the following command in your PyCharm terminal:

```shell
bash docker/psynet debug local
```

It'll print a lot of stuff, but eventually you should see 'Dashboard link' printed.
Open the provided URL in Google Chrome, and it'll take you to the experiment dashboard.
From here you can start a new participant session.

## Configuring your PyCharm Python Console

If you are planning to work with this repository, it will be useful to configure PyCharm to use your experiment's Docker
image. To do this, look for a box in the bottom-right corner of your screen that says 'No interpreter'. Click on this
text and click 'Add interpreter'. Click Docker, and then under 'image name' enter '2022-consonance-carillon:latest' (no
quotes). Press OK. Now if you click on 'Python console' at the bottom of the screen, you should get a Python interpreter
corresponding to the image you just built.

## Troubleshooting

### Windows troubleshooting

#### WSL 2 installation is incomplete

If you see a message beginning with "WSL 2 installation is incomplete", you probably need to do the following:

- Click on the link it gives you
- Click on the link under 'download the latest package', open and run the installer once it has downloaded
- Continue with the next steps of the installation
- Note: if you run Powershell, it might fail if you run it on admin mode! If you get stuck (Access Denied),
  try running it again without admin mode and see if it works.

#### Hardware assisted virtualization

If you see a message beginning "Hardware assisted virtualization and data execution protection must be enabled in the
BIOS", you need to restart your computer into BIOS and change some settings to enable those two things. The precise set
of steps will depend on your computer. The first step though is to restart your computer, and press a certain key to
launch into BIOS -- ordinarily that key will be printed on the screen at some point during the startup sequence.
Hint -- you might find that the option you need to select is called 'SVM mode'...

#### Failed to solve with frontend dockerfile

If you see a message starting "failed to solve with frontend dockerfile.v0",
you may want to try rebooting your computer and trying again.

#### Invalid option name: pipefail

If you see an error message like this when running a Docker command:

```
: command not found 2: 
: command not found 4: 
: invalid option name: set: pipefail
```

The problem is probably that your project has the wrong line endings;
on Windows, if you are not configured correctly, then your files may end up
with Windows-style line endings (CRLF) instead of Unix-style line endings (LF).
To fix this, first follow the line-endings instructions described above for 
setting up Git and PyCharm in Windows.
Then select your project folder in the project pane, 
and from the task bar select File | File Properties | Line Separators | LF - Unix and MacOS.
Your command should now run without the error.

#### A timeout occurred

When starting Docker for Windows you might run into following error: "A timeout occured while waiting for a
WSL integration agent to become ready". In that case, you may want to try installing
an older version of Docker Desktop (e.g. 4.17.1).
