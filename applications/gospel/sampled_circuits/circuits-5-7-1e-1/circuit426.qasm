OPENQASM 2.0;
include "qelib1.inc";
qreg q427[5];
rx(3*pi/2) q427[4];
cx q427[4],q427[3];
cx q427[3],q427[2];
cx q427[2],q427[1];
cx q427[0],q427[1];
