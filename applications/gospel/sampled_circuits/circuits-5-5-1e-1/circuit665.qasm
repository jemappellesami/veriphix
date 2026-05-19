OPENQASM 2.0;
include "qelib1.inc";
qreg q666[5];
cx q666[4],q666[3];
cx q666[3],q666[2];
cx q666[1],q666[2];
cx q666[0],q666[1];
