OPENQASM 2.0;
include "qelib1.inc";
qreg q673[4];
rx(3*pi/4) q673[3];
cx q673[2],q673[3];
cx q673[2],q673[1];
cx q673[1],q673[0];
