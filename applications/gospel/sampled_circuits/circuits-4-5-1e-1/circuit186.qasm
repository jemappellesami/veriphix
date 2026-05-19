OPENQASM 2.0;
include "qelib1.inc";
qreg q187[4];
cx q187[0],q187[1];
rx(7*pi/4) q187[3];
cx q187[2],q187[3];
cx q187[1],q187[2];
cx q187[1],q187[0];
