OPENQASM 2.0;
include "qelib1.inc";
qreg q810[6];
cx q810[0],q810[1];
cx q810[3],q810[2];
rz(3*pi/2) q810[1];
cx q810[1],q810[2];
cx q810[1],q810[0];
