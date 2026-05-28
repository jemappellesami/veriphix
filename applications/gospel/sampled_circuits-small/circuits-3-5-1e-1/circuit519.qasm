OPENQASM 2.0;
include "qelib1.inc";
qreg q520[3];
cx q520[1],q520[0];
rz(pi) q520[2];
rx(5*pi/4) q520[0];
cx q520[1],q520[2];
cx q520[1],q520[0];
