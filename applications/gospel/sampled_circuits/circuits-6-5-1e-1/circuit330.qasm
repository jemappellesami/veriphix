OPENQASM 2.0;
include "qelib1.inc";
qreg q331[6];
cx q331[5],q331[4];
cx q331[3],q331[4];
cx q331[2],q331[3];
cx q331[1],q331[2];
cx q331[0],q331[1];
