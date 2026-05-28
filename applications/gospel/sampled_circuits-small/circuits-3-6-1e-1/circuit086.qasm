OPENQASM 2.0;
include "qelib1.inc";
qreg q87[3];
rx(3*pi/4) q87[0];
rx(pi/2) q87[2];
cx q87[2],q87[1];
cx q87[1],q87[0];
rx(pi/4) q87[1];
