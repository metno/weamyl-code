#!/bin/sh
URL="http://127.0.0.1:8443/thredds/catalog/mepslatest/catalog.xml"
DRAIN_SECONDS=300

function drain() {
    iptables -A INPUT -i lo -j ACCEPT
    iptables -A INPUT -p tcp -m state --state RELATED,ESTABLISHED -j ACCEPT
    iptables -A INPUT -p tcp --dport 8443 -j REJECT
    iptables -A INPUT -p tcp --dport 8080 -j REJECT
}

function release() {
    iptables -F INPUT
}

function start_all() {
    systemctl start telegraf
    systemctl start sensu-client
    systemctl start thredds-reinit-topaz
    systemctl start jhealthz.timer
}

set -e
set -x

drain
sleep ${DRAIN_SECONDS}
systemctl restart thredds
sleep 10
echo "Waiting for service to become available..."
OK="FAILED"
for i in $(seq 120); do
    HTTP_CODE=$(curl -I -sS "${URL}" 2>/dev/null | head -n1 | grep "^HTTP/" | awk '{print $2}')
    if [ "${HTTP_CODE}" == "200" ]; then
      OK="OK"
      break
    fi
    sleep 30
done

if [ "${OK}" == "OK" ]; then
    start_all
    release
fi
