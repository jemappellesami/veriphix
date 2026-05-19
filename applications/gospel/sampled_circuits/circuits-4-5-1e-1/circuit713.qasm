OPENQASM 2.0;
include "qelib1.inc";
qreg q714[4];
rx(pi/2) q714[3];
cx q714[3],q714[2];
cx q714[2],q714[1];
cx q714[0],q714[1];
