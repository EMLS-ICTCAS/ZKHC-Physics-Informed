#add colors to echo's
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
GREEN='\033[1;32m'
RED='\033[1;31m'
RESET='\033[0m'

echo -e "${YELLOW}Compiling...${RESET}"
make
if [ $? -ne 0 ]; then
  echo -e "${RED}Compilation failed!${RESET}"
  exit 1
fi
echo -e "${GREEN}Compilation done!${RESET}"



echo -e "${YELLOW}Running...${RESET}"
# check if the folder output/results exists, if not create it
if [ ! -d "output/results" ]; then
  mkdir -p output/results
else
  # remove the files inside output/results
  rm -rf output/results/*
fi
rm -rf output/*.h5
./bin/main
if [ $? -ne 0 ]; then
  echo -e "${RED}Running failed!${RESET}"
  exit 1
fi
echo -e "${GREEN}Running done!${RESET}"

echo -e "${YELLOW}Animating...${RESET}"
# use python3 if you have it installed, otherwise use python
if [ -x "$(command -v python3)" ]; then
  python3 src/animation.py
else
  python src/animation.py
fi
if [ $? -ne 0 ]; then
  echo -e "${RED}Animating failed!${RESET}"
  exit 1
fi
echo -e "${GREEN}Animating done!${RESET}"
