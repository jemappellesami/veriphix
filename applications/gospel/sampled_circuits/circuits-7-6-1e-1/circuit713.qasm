OPENQASM 2.0;
include "qelib1.inc";
qreg q714[7];
cx q714[3],q714[4];
cx q714[3],q714[2];
cx q714[1],q714[2];
cx q714[0],q714[1];
rx(pi/4) q714[1];
