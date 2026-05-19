OPENQASM 2.0;
include "qelib1.inc";
qreg q325[5];
cx q325[4],q325[3];
cx q325[2],q325[3];
cx q325[2],q325[1];
cx q325[1],q325[0];
