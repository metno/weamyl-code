iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -p tcp -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A INPUT -p tcp --dport 8443 -j REJECT
iptables -A INPUT -p tcp --dport 8080 -j REJECT
