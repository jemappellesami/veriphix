OPENQASM 2.0;
include "qelib1.inc";
qreg q296[3];
rx(7*pi/4) q296[0];
cx q296[0],q296[1];
cx q296[1],q296[2];
cx q296[1],q296[0];
rx(pi/4) q296[1];
