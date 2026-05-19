OPENQASM 2.0;
include "qelib1.inc";
qreg q204[5];
cx q204[4],q204[3];
cx q204[3],q204[2];
cx q204[1],q204[2];
cx q204[0],q204[1];
