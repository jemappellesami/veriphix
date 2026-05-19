OPENQASM 2.0;
include "qelib1.inc";
qreg q245[3];
rx(5*pi/4) q245[2];
rz(pi/2) q245[2];
cx q245[2],q245[1];
cx q245[0],q245[1];
rx(pi/4) q245[1];
