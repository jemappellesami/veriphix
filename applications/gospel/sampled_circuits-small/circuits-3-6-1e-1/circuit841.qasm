OPENQASM 2.0;
include "qelib1.inc";
qreg q842[3];
cx q842[1],q842[0];
rx(5*pi/4) q842[1];
cx q842[2],q842[1];
cx q842[0],q842[1];
rx(pi/4) q842[1];
