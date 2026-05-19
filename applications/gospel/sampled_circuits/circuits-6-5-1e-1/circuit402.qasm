OPENQASM 2.0;
include "qelib1.inc";
qreg q403[6];
cx q403[4],q403[5];
cx q403[3],q403[4];
cx q403[2],q403[3];
cx q403[2],q403[1];
cx q403[1],q403[0];
