#!/bin/bash
INTERVAL="$1"
WATCH_PATH="$2"

function usage {
    echo "Usage: $0 sleep_seconds file_path_to_monitor_for_changed_mtime" >>/dev/stderr
    exit 1
}

if [ -z "${INTERVAL}" ]; then
    echo "No interval provided" >>/dev/stderr
    usage
fi

if [ ! -e "${WATCH_PATH}" ]; then
    echo ""${WATCH_PATH}" does not exist." >>/dev/stderr
    usage
fi

function watch {
    CUR=$(/usr/bin/stat -L -c %Z "${WATCH_PATH}")
}
watch
PREV="${CUR}"

while true; do
    sleep "${INTERVAL}" || usage
    watch
    if [ "${PREV}" -ne "${CUR}" ]; then
        PREV="${CUR}"
        ################################################## 
        /usr/bin/docker exec thredds /usr/local/tomcat/bin/reinit-thredds.sh
        ##################################################
    fi
done
