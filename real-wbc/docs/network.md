# Network Setup for Fully Untethered Development 
The network setup procedure should be the same for most of the Unitree robots, not only Go2. Having a good internet connection will be extremely beneficial for reproducing our robot pipeline (attaching VSCode editors, downloading checkpoints, building docker containers, etc.). We highly recommend carefully setting-up the wireless internet connection to get fully untethered deployment.

## Local Network Configuration
Unitree robots use a same local network segment: `192.168.123.xxx`. Thus, after using an ethernet cable to connect your computer to a unitree robot, you need to set your ip address to the same segment.

Please check out the unitree official [documentation](https://support.unitree.com/home/en/developer/ROS2_service) and follow the *Connect to Unitree robot* section.



## Internet Connection

Since the net port on the unitree jetson is already assigned with the local network segment, directly connecting it to a router or a net port in your office may not work. To have internet access, you can either use a usb-ethernet adapter or a usb-wifi module.

After connecting a usb-ethernet adapter to your Unitree jetson and then use an ethernet cable to connect it to the internet net port (in your lab/office), it may still unable to connect to the internet. This is because the default route with the highest priority is the local network: after typing `ip a`, you may find `default via 192.168.123.1` on top of all other routes. You can delete this route so that the other route (through the adapter to the internet) has the highest priority:

```sh
sudo route del -net 0.0.0.0 netmask 0.0.0.0 gw 192.168.123.1
```

To run this code automatically every time you turn on your robot, you can create a system service that automatically runs it.

First write a script and save it to `/home/unitree/delete_route.sh`.

```sh
#!/bin/bash

while true;
   do
    if ip route | head -n 1 | grep -q "default via 192.168.123.1"; then
        # Delete the route if it exists
        sudo route del -net 0.0.0.0 netmask 0.0.0.0 gw 192.168.123.1
    fi 
    sleep 10s
   done
```

Then create this service file and save it into the systemd directory `/etc/systemd/system/delete_route.service`
```
[Unit]
Description=Disable the default LAN route 
After=network.target

[Service]
ExecStart=/home/unitree/delete_route.sh
type=simple
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```


Finally enable the system service

```sh
sudo systemctl daemon-reload
sudo systemctl enable delete_route.service
sudo systemctl start delete_route.service
```

## Wifi Module Setup

We tested two wifi modules:

- [A USB2.0 module](https://a.co/d/0ciXDmGC): Low speed, but doesn't require additional driver.
- [A USB3.0 module](https://a.co/d/0hogevhr): High speed, but requires additional driver

We recommend that you buy both modules because you may need internet to install the driver for the faster module. 

Note that it may take some time (~1min) for the wifi module to be recognized by the linux system. Therefore, it is normal if the wifi is not connected right after the system boots up.

### AC1300 Wifi driver installation guide

If you already have Internet access on your Jetson module, directly clone the following repository. If not, you can download it to your laptop and use the local network to synchronize the repository to Jetson.
```sh
git clone https://github.com/cilynx/rtl88x2bu.git
cd rtl88x2bu
vim Makefile
```
Then configure the target platform in the `Makefile`: find the following config options 
```Makefile
CONFIG_PLATFORM_I386_PC = y
...
CONFIG_PLATFORM_ARM_NV_NANO = n
```
and modify it to
```Makefile
CONFIG_PLATFORM_I386_PC = n
...
CONFIG_PLATFORM_ARM_NV_NANO = y
```
The driver works for other NV Jetson modules as well (not only Jetson Nano).

Finally, build the driver and install it to your jetson:
```sh
make -j
sudo make install
sudo modprobe 88x2bu
```
Reboot, then connect your TP-link AC1300 adapter. Wifi connection should be available in ~1 minute.

Reference: https://forums.developer.nvidia.com/t/wifi-adapter-tplink-ac1300/243480

## Trouble Shooting

### Q1: The wifi module works under my personal hotspot, but I have trouble connecting it to the university wifi.
This happens when there is an authentication process to connect to a university wifi (e.g. Stanford). You can first follow the [previous section](#internet-connection) and make sure the default route is correct. If the authentication page still doesn't show up, you can try to connect your wifi adapter to your own computer where you should be able to successfully register it to your university network. Make sure you have turned off all the other network interfaces in your computer and the wifi adapter is the only valid interface.

