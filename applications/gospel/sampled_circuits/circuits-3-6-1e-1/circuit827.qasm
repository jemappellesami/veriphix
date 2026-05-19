OPENQASM 2.0;
include "qelib1.inc";
qreg q828[3];
rz(3*pi/4) q828[2];
cx q828[2],q828[1];
cx q828[0],q828[1];
rx(pi/4) q828[1];
