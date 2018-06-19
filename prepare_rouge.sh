#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p $CURRENT_DIR/src
cd $CURRENT_DIR/src && git clone https://github.com/kylehg/summarizer.git || true
cd $CURRENT_DIR
export ROUGE=$CURRENT_DIR/src/summarizer/rouge/ROUGE-1.5.5
