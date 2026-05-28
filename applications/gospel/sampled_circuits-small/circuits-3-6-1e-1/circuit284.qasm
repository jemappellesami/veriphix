OPENQASM 2.0;
include "qelib1.inc";
qreg q285[3];
rx(3*pi/2) q285[2];
cx q285[1],q285[2];
cx q285[1],q285[0];
rx(pi/4) q285[1];
