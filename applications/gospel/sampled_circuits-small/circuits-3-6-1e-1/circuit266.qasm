OPENQASM 2.0;
include "qelib1.inc";
qreg q267[3];
rx(3*pi/4) q267[2];
cx q267[2],q267[1];
cx q267[0],q267[1];
rx(pi/4) q267[1];
