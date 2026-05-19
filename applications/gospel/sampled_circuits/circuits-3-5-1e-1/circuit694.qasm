OPENQASM 2.0;
include "qelib1.inc";
qreg q695[3];
cx q695[1],q695[0];
rx(7*pi/4) q695[2];
rz(pi) q695[0];
cx q695[2],q695[1];
cx q695[1],q695[0];
