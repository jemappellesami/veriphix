OPENQASM 2.0;
include "qelib1.inc";
qreg q687[3];
cx q687[0],q687[1];
rx(7*pi/4) q687[2];
cx q687[2],q687[1];
cx q687[0],q687[1];
