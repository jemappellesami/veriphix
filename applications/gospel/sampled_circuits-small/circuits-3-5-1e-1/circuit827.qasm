OPENQASM 2.0;
include "qelib1.inc";
qreg q828[3];
rx(5*pi/4) q828[2];
rz(pi/4) q828[2];
cx q828[1],q828[2];
cx q828[1],q828[0];
