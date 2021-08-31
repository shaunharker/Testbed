#!/bin/bash
ssh -CNL localhost:8000:localhost:8000 $REMOTEJUPYTERIP &
