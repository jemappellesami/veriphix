OPENQASM 2.0;
include "qelib1.inc";
qreg q421[6];
cx q421[4],q421[5];
cx q421[3],q421[4];
cx q421[2],q421[3];
cx q421[2],q421[1];
cx q421[0],q421[1];
