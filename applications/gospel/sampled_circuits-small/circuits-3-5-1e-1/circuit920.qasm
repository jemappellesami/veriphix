OPENQASM 2.0;
include "qelib1.inc";
qreg q921[3];
rx(3*pi/4) q921[0];
rz(pi) q921[0];
cx q921[1],q921[2];
rx(3*pi/4) q921[1];
cx q921[1],q921[0];
