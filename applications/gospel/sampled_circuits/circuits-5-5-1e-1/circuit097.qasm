OPENQASM 2.0;
include "qelib1.inc";
qreg q98[5];
rx(7*pi/4) q98[0];
cx q98[4],q98[3];
cx q98[0],q98[1];
cx q98[1],q98[2];
cx q98[1],q98[0];
