OPENQASM 2.0;
include "qelib1.inc";
qreg q520[4];
rz(7*pi/4) q520[3];
cx q520[3],q520[2];
cx q520[2],q520[1];
cx q520[1],q520[0];
rx(pi/4) q520[1];
