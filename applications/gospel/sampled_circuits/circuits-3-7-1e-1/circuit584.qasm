OPENQASM 2.0;
include "qelib1.inc";
qreg q585[3];
cx q585[1],q585[2];
rz(5*pi/4) q585[1];
rz(3*pi/4) q585[2];
cx q585[2],q585[1];
cx q585[1],q585[0];
