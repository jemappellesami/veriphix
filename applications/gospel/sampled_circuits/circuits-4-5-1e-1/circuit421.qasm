OPENQASM 2.0;
include "qelib1.inc";
qreg q422[4];
cx q422[1],q422[0];
rx(pi) q422[1];
rx(3*pi/4) q422[0];
cx q422[1],q422[2];
cx q422[0],q422[1];
