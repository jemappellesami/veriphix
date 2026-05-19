OPENQASM 2.0;
include "qelib1.inc";
qreg q888[3];
cx q888[0],q888[1];
rx(pi/2) q888[1];
cx q888[1],q888[0];
cx q888[2],q888[1];
rx(pi/4) q888[0];
