OPENQASM 2.0;
include "qelib1.inc";
qreg q852[4];
rx(3*pi/4) q852[3];
cx q852[3],q852[2];
cx q852[1],q852[2];
cx q852[0],q852[1];
rx(pi/4) q852[1];
