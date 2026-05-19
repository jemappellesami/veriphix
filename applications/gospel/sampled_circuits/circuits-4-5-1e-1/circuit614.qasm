OPENQASM 2.0;
include "qelib1.inc";
qreg q615[4];
rz(pi/4) q615[3];
cx q615[2],q615[3];
cx q615[2],q615[1];
cx q615[0],q615[1];
