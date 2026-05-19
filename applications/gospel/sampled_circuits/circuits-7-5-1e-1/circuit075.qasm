OPENQASM 2.0;
include "qelib1.inc";
qreg q76[7];
cx q76[4],q76[5];
cx q76[3],q76[4];
cx q76[3],q76[2];
cx q76[2],q76[1];
cx q76[0],q76[1];
