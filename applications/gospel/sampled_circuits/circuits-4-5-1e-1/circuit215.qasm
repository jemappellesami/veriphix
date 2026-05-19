OPENQASM 2.0;
include "qelib1.inc";
qreg q216[4];
rx(3*pi/4) q216[3];
rz(pi) q216[3];
cx q216[3],q216[2];
cx q216[2],q216[1];
cx q216[1],q216[0];
