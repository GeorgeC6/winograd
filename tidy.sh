#!/bin/bash

# start
echo "Starting cleanup process..."

echo "Removing log and error files..."
rm -f ./*.log ./*.err ./*.out

# end
echo "Tidying done."
