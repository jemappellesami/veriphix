OPENQASM 2.0;
include "qelib1.inc";
qreg q325[7];
cx q325[4],q325[5];
cx q325[4],q325[3];
cx q325[3],q325[2];
cx q325[1],q325[2];
cx q325[1],q325[0];
