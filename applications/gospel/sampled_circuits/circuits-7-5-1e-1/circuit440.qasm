OPENQASM 2.0;
include "qelib1.inc";
qreg q441[7];
cx q441[4],q441[5];
cx q441[4],q441[3];
cx q441[3],q441[2];
cx q441[1],q441[2];
cx q441[1],q441[0];
