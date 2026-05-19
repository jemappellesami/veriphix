OPENQASM 2.0;
include "qelib1.inc";
qreg q63[4];
rz(3*pi/4) q63[2];
cx q63[3],q63[2];
cx q63[2],q63[1];
cx q63[0],q63[1];
rx(pi/4) q63[1];
