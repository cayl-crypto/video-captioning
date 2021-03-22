#!/usr/bin/env sh
# This script downloads the dataset MSVD.
# Skip this if you already have the the dataset MSVD.

echo "Downloading video captioning dataset MSVD - Youtube Clips [~1.7GB] ..."

DIR="./"

if [ ! -d "$DIR" ]; then
    mkdir $DIR
fi

FILENAME=YouTubeClips.tar

wget --no-check-certificate https://www.cs.utexas.edu/users/ml/clamp/videoDescription/$FILENAME

echo "Unzipping "

tar -xf $FILENAME && rm -f $FILENAME

echo "Done."
