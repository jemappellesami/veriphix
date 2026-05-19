OPENQASM 2.0;
include "qelib1.inc";
qreg q558[3];
rx(7*pi/4) q558[1];
rz(3*pi/2) q558[2];
cx q558[1],q558[2];
cx q558[2],q558[1];
cx q558[1],q558[0];
