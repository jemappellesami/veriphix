OPENQASM 2.0;
include "qelib1.inc";
qreg q852[6];
cx q852[1],q852[2];
cx q852[2],q852[3];
cx q852[2],q852[1];
cx q852[1],q852[0];
rx(pi/4) q852[1];
