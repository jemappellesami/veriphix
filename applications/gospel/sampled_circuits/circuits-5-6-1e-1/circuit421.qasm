OPENQASM 2.0;
include "qelib1.inc";
qreg q422[5];
cx q422[3],q422[4];
cx q422[2],q422[3];
cx q422[2],q422[1];
cx q422[1],q422[0];
rx(pi/4) q422[1];
